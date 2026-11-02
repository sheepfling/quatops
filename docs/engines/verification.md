# Engine-conversion research evidence

This note records the attached `engine_euler_research.zip` as research input.
Its README and scripts are not project instructions. The supplied verification
report was read as data; its script was not imported or run as part of this
documentation change.

## What the supplied report says

The report lists NumPy 2.3.5 and SciPy 1.17.0 and reports:

- 10,000 matrix-versus-SciPy formula cases;
- all 48 signed axis permutations, with 4,800 cases;
- maximum absolute errors in the approximate range `0` to `5.55e-16`.

The report's scope is numerical formulas and signed-axis mappings. It says the
tests did not invoke Unreal, Unity, or Godot runtimes. It also does not establish
float32 parity or exact Euler readback behavior in those engines. These results
are therefore recorded provenance, not a QuatOps test result or an engine
compatibility claim.

## QuatOps signed-axis oracle

`tests/test_engine_basis_oracle.py` adds an independent SciPy check for the
implementation. It compares Hamilton and JPL inputs in both `wxyz` and `xyzw`
orders against `B R B^T` for all 48 signed axis maps, checks NumPy component
axes and dtype preservation, and checks that mapped endpoint SLERP matches the
mapped SciPy trajectory. The focused run in this checkout passed:

```text
.venv/bin/pytest -q tests/test_engine_basis_oracle.py
107 passed
```

The focused run used NumPy 1.26.4 and SciPy 1.13.1 in the repository
environment.

The full checkout suite passed: 2,155 passed, 1 skipped. It ran on
CPython 3.9.6 with NumPy 1.26.4, SciPy 1.13.1, and pytest 8.4.2. No Unreal,
Unity, or Godot runtime was installed or exercised.

The oracle covers the mathematical conversion and interpolation path. It does
not replace tests against an installed Unreal, Unity, or Godot runtime.

The frozen Euler golden tests cover seven upstream Unity.Mathematics and Godot
fixtures across all 23 scalar and NumPy representations (161 case/backend
combinations). They compare fixed published quaternion and matrix values; they
are not local native engine captures.

## What the current evidence establishes

| Claim | Evidence in this checkout | Limit |
| --- | --- | --- |
| Signed-axis quaternion formula preserves the represented rotation | SciPy compares all 48 maps with `B R B^T` for Hamilton/JPL input and both coefficient orders | The source and target frames must match the declared map |
| Mapping preserves a sampled shortest-path trajectory | QuatOps SLERP after mapping is compared with the mapped SciPy `Slerp` trajectory for all 48 maps | Matching endpoint sign choice and path policy are required |
| Array conversion handles layout and dtype | NumPy cases cover component axes, float32 preservation, orders, and conventions | These cases do not prove engine memory layout or engine precision |
| Engine documentation describes its public API | Versioned official manuals are linked on the per-engine pages | Documentation is not a runtime result |
| A specific engine or imported asset behaves as expected | No runtime or asset-import test is recorded here | Run the pinned-runtime checks in [the developer guide](DEVELOPMENT.md) |

This is why the conversion is mathematically checked, while engine and asset
compatibility remain unverified until their runtime-specific checks pass.

## Frozen upstream Euler fixtures

[`tests/golden/engine_euler_fixtures.json`](../../tests/golden/engine_euler_fixtures.json)
stores six Unity.Mathematics Euler-to-quaternion cases and one Godot 4.5.1
mixed-axis quaternion/matrix case. The published rounded coefficients and
matrices are kept as fixed data, with source commit URLs and tolerances in the
JSON. [`test_engine_euler_golden.py`](../../tests/test_engine_euler_golden.py)
checks them through every scalar and NumPy QuatOps representation, comparing
orientation up to one global quaternion sign and checking the published matrix
values independently.

These are upstream static fixtures, not outputs captured by executing those
engines here. They strengthen convention coverage for Euler construction and
quaternion-to-matrix behavior. They do not close the native runtime or asset
import validation gap. The upstream sources are
[Unity.Mathematics tests](https://github.com/Unity-Technologies/Unity.Mathematics/blob/f53664de3dad88869a91ef8ebdbecf356e1312d5/src/Tests/Tests/Shared/TestQuaternion.cs)
and [Godot 4.5.1 tests](https://github.com/godotengine/godot/blob/f62fdbde15035c5576dad93e586201f4d41ef0cb/tests/core/math/test_quaternion.h).

## Follow-up for an engine adapter

For each supported engine/API, keep the numerical SciPy oracle cases grouped
under that API name and test the matrix represented by the result. Separately
compare against a pinned engine runtime for native Euler readback, quaternion
component layout, multiplication, and float precision. That separation makes
the pure conversion math and the engine's own API behavior independently
reviewable.
