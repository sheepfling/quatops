# Game-engine references

This directory keeps engine-specific Euler and coordinate-system notes close
to the library documentation without making the core API engine-dependent.
The [API contract](../API.md) defines QuatOps behavior. These pages describe
external APIs and conversion guidance.

Use `quatops.engines.SignedAxisMap` for caller-declared signed-axis conversion
to engine XYZW values. The [trajectory guide](trajectory.md) shows how to
preconvert a key sequence and sample it natively.

For implementation workflow, start with the [developer guide](../contributing/ENGINE_INTEGRATIONS.md).
Coding agents working in this area should follow the
[agent guide](../contributing/ENGINE_AGENT_GUIDE.md); the repository's `AGENTS.md` points here too.

## Choose a reference

| Library/API | Reference | Euler behavior at a glance |
| --- | --- | --- |
| Unreal Engine `FRotator` / `FQuat` | [Unreal](unreal.md) | `FRotator` fields are Pitch, Yaw, Roll in degrees; Unreal's native basis is left-handed, Z-up. |
| Unity `UnityEngine.Quaternion` | [Unity](unity.md) | `Quaternion.Euler(x, y, z)` takes degrees and applies Z, then X, then Y. |
| Unity `Unity.Mathematics.quaternion` | [Unity](unity.md) | Sequence-named constructors take radians and make the order explicit. |
| Godot `Quaternion` / `Basis` / `Node3D` | [Godot](godot.md) | `Quaternion.from_euler` takes radians and uses YXZ; `Basis` exposes order-specific constructors. |

The [coordinate-mapping page](coordinate-mapping.md) keeps the basis-change
formulas in one place. The SciPy conversion oracle lives in
[`tests/test_engine_basis_oracle.py`](../../tests/test_engine_basis_oracle.py).
Native Unreal runtime checks are tracked in
[issue #4](https://github.com/sheepfling/quatops/issues/4); Unity and Godot
checks are tracked in [issue #7](https://github.com/sheepfling/quatops/issues/7).
For the main use case, start with the [trajectory conversion guide](trajectory.md).

## Reading these pages safely

An Euler triple is incomplete without its API, axis order, units, frame
(intrinsic or extrinsic), and rotation direction. “Yaw-pitch-roll” by itself
does not uniquely specify a rotation. See the core [Euler contract](../API.md#attitude-conversions-alignment-statistics-and-random-sampling) for
QuatOps' active/passive definitions and intrinsic/extrinsic sequence rules.

Engine coordinate handedness is a separate issue from Hamilton versus JPL
quaternion algebra. A coordinate-basis change can alter quaternion vector
components without changing the algebra used by the library.

These pages are references, not claims of engine-runtime compatibility. The
linked engine manuals are versioned independently. Before shipping an adapter,
pin its engine/API version and validate it against that runtime. If runtime
helpers are added later, keep them under engine-specific namespaces such as
`quatops.engines.unreal`, `quatops.engines.unity`, and
`quatops.engines.godot`; the generic conversion API can remain unchanged.
