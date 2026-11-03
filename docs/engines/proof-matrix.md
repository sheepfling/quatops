# Quaternion converter evidence matrix

This matrix is scoped to the named quaternion-to-quaternion adapters. It does
not require Euler, animation, hierarchy, or asset-import behavior.

| Target build | Native API | Stored witness set | Conversion status |
| --- | --- | --- | --- |
| Unreal 5.7.4 | `TQuat<T>` construction and `RotateVector`; float32/float64 | [Contract and witnesses](../../tests/oracles/unreal/5.7.4/contract.json) | Native physical-action evidence and bidirectional adapter checks pass |
| Unity 6000.5.2f1 | `UnityEngine.Quaternion` construction and `Quaternion * Vector3`; float32 | [Contract and witnesses](../../tests/oracles/unity/6000.5.2f1/contract.json) | Native physical-action evidence and bidirectional adapter checks pass |
| Godot 4.7 stable | `Quaternion` construction and `Quaternion * Vector3`; float32 | [Contract and witnesses](../../tests/oracles/godot/4.7-stable/contract.json) | Native physical-action evidence and bidirectional adapter checks pass |
| Godot 4.8-dev1 | `Quaternion` construction and `Quaternion * Vector3`; float32 | [Contract and witnesses](../../tests/oracles/godot/4.8-dev1/contract.json) | Native physical-action evidence and bidirectional adapter checks pass |
| Blender 5.1.2 | `mathutils.Quaternion` construction and `Quaternion @ Vector` | [Contract and witnesses](../../tests/oracles/blender/5.1.2/contract.json) | Native physical-action evidence and bidirectional adapter checks pass |

Every retained target includes identity, three positive quarter-turns, two
unequal mixed orientations, one antipode, a nonprincipal half-turn, and both
orders of the mixed product. Each orientation retains its exact canonical and
native coefficients plus the native images of +X, +Y, +Z and one asymmetric
probe. Build identity, native precision, capture-source hash, input hash, and
the original full-capture hash are kept with each version.

The broad signed-axis checks are generated in
[`test_engine_basis_oracle.py`](../../tests/test_engine_basis_oracle.py): all
48 signed-axis maps and all four named adapters are checked without storing
passing randomized outputs. Native fixtures are a small target-contract
witness set, not a duplicate corpus of the shared mathematics checks.

The actual capture runs and SciPy formula checks remain separately identified:
the pinned engine supplies the frozen vector actions; SciPy independently
checks the declared basis equations. See the shared
[`CAP-300 contract`](CAP-300.md) and
[`coordinate-mapping.md`](coordinate-mapping.md).
