# Game-engine references

This directory keeps engine-specific Euler and coordinate-system notes close
to the library documentation without making the core API engine-dependent.
The [API contract](../API.md) defines QuatOps behavior. These pages describe
external APIs and conversion guidance.

Use `quatops.engines.SignedAxisMap` and `QuaternionFrameMap` for caller-declared
frame conversion. The named `quatops.engines.unreal`, `.unity`, `.godot`, and
`.blender` modules encode each target API's quaternion storage order; every
conversion requires a precompiled frame map. They do not assume engine or asset
axes. The [trajectory guide](trajectory.md) shows how to preconvert a key
sequence and sample it natively.

For implementation workflow, start with the [developer guide](DEVELOPMENT.md).
Coding agents working in this area should follow the
[agent guide](AGENT_GUIDE.md); the repository's `AGENTS.md` points here too.

## Choose a reference

| Library/API | Reference | Euler behavior at a glance |
| --- | --- | --- |
| Unreal Engine `FRotator` / `FQuat` | [Unreal](unreal.md) | `FRotator` fields are Pitch, Yaw, Roll in degrees; Unreal's native basis is left-handed, Z-up. |
| Unity `UnityEngine.Quaternion` | [Unity](unity.md) | `Quaternion.Euler(x, y, z)` takes degrees and applies Z, then X, then Y. |
| Unity `Unity.Mathematics.quaternion` | [Unity](unity.md) | Sequence-named constructors take radians and make the order explicit. |
| Godot `Quaternion` / `Basis` / `Node3D` | [Godot](godot.md) | `Quaternion.from_euler` takes radians and uses YXZ; `Basis` exposes order-specific constructors. |
| Blender `mathutils.Euler` / `Quaternion` / `Matrix` | [Blender](blender.md) | Radian inputs, WXYZ quaternions, six Euler orders, and reference-aware readback. |

The [coordinate-mapping page](coordinate-mapping.md) keeps the basis-change
formulas in one place. The [verification note](verification.md) separates the
research bundle, SciPy oracle checks, and pinned native engine captures. For
the main use case, start with the [trajectory conversion guide](trajectory.md).
The converter-specific acceptance plan and current bidirectional proof matrix
are in [CAP-300](CAP-300.md).

## Reading these pages safely

An Euler triple is incomplete without its API, axis order, units, frame
(intrinsic or extrinsic), and rotation direction. “Yaw-pitch-roll” by itself
does not uniquely specify a rotation. See the core [Euler contract](../API.md#attitude-conversions-alignment-statistics-and-random-sampling) for
QuatOps' active/passive definitions and intrinsic/extrinsic sequence rules.

Engine coordinate handedness is a separate issue from Hamilton versus JPL
quaternion algebra. A coordinate-basis change can alter quaternion vector
components without changing the algebra used by the library.

These pages describe engine APIs and the runtime evidence available for them.
Versioned native captures cover Unreal Engine 5.7.4, Unity Editor 6000.5.2f1,
Godot 4.7 stable and 4.8-dev1, and Blender 5.1.2. Their shared proof
categories include Euler branch samples, matrix-to-quaternion, axis-angle,
rotation-vector, shortest SLERP, normalization edges, basis/vector action,
composition, and an explicitly declared source-frame mapping. Unreal and Godot
also capture native full-path SLERP; the selected Unity and Blender public APIs
do not expose that mode. Unity adds all six Unity.Mathematics Euler
constructor/readback orders and clamped/unclamped linear interpolation proofs.
Unity rotation-vector results are derived through native axis-angle calls.
Both Godot captures probe all six `Basis` Euler orders; the Godot 4.8-dev1
fixture is a development snapshot. Supplemental Godot from-to
vector-constructor probes are documented separately from the common proof
categories. Blender adds swing-twist, rotation-difference, and Euler-continuity
trajectories; its native uppercase Euler labels were verified against SciPy's
lowercase extrinsic sequences. None of these captures tests imported assets.
See the [proof matrix](proof-matrix.md) for coverage by API route and the
[verification note](verification.md) for current run results. Before shipping
an adapter, pin its engine/API version and validate the remaining integration
paths against that runtime.
