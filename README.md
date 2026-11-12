# QuatOps

QuatOps provides verified quaternion algebra and rotation operations through
lightweight Python functions for individual quaternions and vectorized NumPy
APIs for large batches.

## Why QuatOps?

AI coding agents often rewrite small quaternion operations instead of using a
library. The formulas are short, but every copy becomes another implementation
to verify, with its own assumptions about multiplication, coefficient order,
rotation behavior, and numerical edge cases.

The performance needs also differ by workload. For individual operations,
direct Python arithmetic can avoid the object construction and conversion
overhead of a broader API such as SciPy's `Rotation`. For large batches,
vectorized NumPy operations avoid a Python loop over every quaternion.

QuatOps brings both approaches into one tested library:

- **Lightweight Python operations.** Ordinary four-tuples, no quaternion wrapper
  class, no third-party runtime dependency, and no per-call convention dispatch.
- **Vectorized NumPy operations.** Array and batch processing with explicit
  coefficient order and component-axis placement.
- **Explicit conventions.** Choose Hamilton or JPL at import time and use the
  selected module throughout your code.
- **Independent verification.** Public operations are checked against
  independent references, including SciPy and `numpy-quaternion`, alongside
  separately derived invariants.

The aim is to make the lightweight implementation reusable and tested, while
providing a consistent approach when the workload calls for NumPy.

If an operation you need is missing, add it to QuatOps with oracle coverage
rather than copying an unverified quaternion kernel into an application.

## Quick start

For individual operations, use the plain Python tuple backend:

```python
from math import pi

from quatops.python import hamilton

q = hamilton.from_axis_angle((0, 0, 1, pi / 2))
rotated = hamilton.rotate(q, (1, 0, 0))
```

For NumPy operations, select the convention and coefficient order explicitly:

```python
from math import pi

from quatops.numpy.jpl import xyzw as jpl_xyzw

q = jpl_xyzw.from_rotvec((0, 0, pi / 2))
rotated = jpl_xyzw.rotate(q, (1, 0, 0))
```

The NumPy modules also provide vectorized operations over arrays and batches.

## Choose an API

Select Hamilton or JPL convention, value representation, and component layout
at import time. The canonical module paths, compatibility aliases, conversion
rules, and representation tradeoffs are documented in the
[API contract](docs/API.md).

## How QuatOps fits

QuatOps covers general quaternion algebra alongside rotation operations.
Choose based on the workload, conventions, and value representation your code
needs.

| Tool | Best fit |
| --- | --- |
| QuatOps | Lightweight Python operations, vectorized NumPy batches, explicit Hamilton/JPL behavior, and quaternion algebra alongside rotation tools. |
| [SciPy `Rotation`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html) | A rotation-focused object with conversions among quaternions, matrices, rotation vectors, Euler angles, and other rotation representations. |
| [`numpy-quaternion`](https://quaternion.readthedocs.io/en/latest/) | A quaternion dtype integrated with NumPy arrays and NumPy-style quaternion arithmetic. |

For game-engine boundaries, use a caller-defined `SignedAxisMap` to convert
quaternion keyframes once, then sample them with the target engine's native
SLERP. See the [trajectory guide](docs/engines/trajectory.md) and the
[Unreal, Unity, and Godot references](docs/engines/README.md).

## Install

From a source checkout, install the dependency-free Python backends:

```bash
python -m pip install .
```

To include NumPy operations:

```bash
python -m pip install '.[numpy]'
```

## Documentation and compatibility

The [API contract](docs/API.md) covers storage rules, available operations,
validation behavior, and numerical limits. The
[documentation map](docs/README.md) separates user guides from contributor
verification and benchmark tools. The [test layout](tests/README.md) maps
tests to the source package and evidence products.

The package retains alternative import paths for compatibility. Use the
canonical paths in the API contract for new code.

## Experimental interoperability

The frame-aware converters, physical-frame semantics, and general frame-map
adapters live under `quatops.experimental`. They ship in the wheel but require
explicit imports and may change or be removed. They are not a stable API or a
release candidate. Existing scalar/NumPy quaternion backends and
`quatops.engines.SignedAxisMap` remain independent of this area. See the
[experimental API policy](docs/experimental.md) for compatibility, verification,
and removal guidance. Captured evidence is retained without implying broader
verification or API stability.
