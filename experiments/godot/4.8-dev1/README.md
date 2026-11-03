# Godot 4.8-dev1 capture

This headless GDScript capture uses the installed official Windows build at
commit `ebbf577a041d1ab8d7824b6f90e7d0575461ca45`. It records the engine
version and uses float32 values.

From the repository root in PowerShell, regenerate the native fixture:

```powershell
$root = (Resolve-Path .).Path
$godot = 'C:\Users\Public\Godot\engines\windows\Godot_v4.8-dev1_win64.exe\Godot_v4.8-dev1_win64_console.exe'
$project = Join-Path $root 'experiments\godot\4.8-dev1'
$output = Join-Path $root 'tests\golden\godot\4.8-dev1\native_rotation.json'
& $godot --headless --path $project --script res://capture.gd -- $output
if ($LASTEXITCODE -ne 0) { throw "Godot capture failed with exit code $LASTEXITCODE" }
```

The native capture records fifteen default `Quaternion.from_euler` cases and
67 shared triples for each of all six `Basis.from_euler` orders. It records
both order-specific Euler readbacks and their roundtrips, including values
around positive and negative 90 degrees and adjacent float32 values. The
extended capture adds ten valid matrix-to-quaternion inputs and three
native-only near-collinear basis diagnostics, axis-angle and rotation-vector
routes, explicit `Quaternion.exp`/`log` branches, nine `Quaternion(from, to)`
cases, 22 shortest and no-shortest-path SLERP samples, normalization edges,
basis action, cross products, quaternion and `Transform3D` composition, and a
declared source-frame map. The direct `Quaternion(Basis)` calls on diagnostic
inputs intentionally emit Godot's orthonormality error and record the returned
identity result; those inputs are not passed to QuatOps. A non-finite native
result is represented as JSON `null` and paired with an `is_finite` field.

The fixture's `canonical_interop` section adds direct native basis and
mixed-vector actions for the shared WXYZ inputs and a noncommuting product.

Run the separate QuatOps/SciPy comparisons with:

```powershell
python -m pytest tests\test_godot_native_fixtures.py tests\test_native_engine_extended_conversions.py tests\test_canonical_interoperability.py
```

Godot `slerpni` is captured as the engine's no-shortest-path route. This
fixture covers this development snapshot only; the separate 4.7 stable capture
is stored beside it. Neither capture tests imported assets, nonuniform-scale
shear, or a double-precision build.
