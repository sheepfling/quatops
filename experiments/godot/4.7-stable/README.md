# Godot 4.7 stable capture

This headless GDScript capture uses the installed official Windows Godot 4.7
stable build. It records the runtime version, build hash, platform, and
float32 precision in the fixture.

From the repository root in PowerShell, regenerate the native fixture:

```powershell
$root = (Resolve-Path .).Path
$godot = 'C:\Users\Public\Godot\engines\windows\Godot_v4.7-stable_win64.exe\Godot_v4.7-stable_win64_console.exe'
$project = Join-Path $root 'experiments\godot\4.7-stable'
$output = Join-Path $root 'tests\golden\godot\4.7-stable\native_rotation.json'
& $godot --headless --path $project --script res://capture.gd -- $output
if ($LASTEXITCODE -ne 0) { throw "Godot capture failed with exit code $LASTEXITCODE" }
```

The capture records all six `Basis.from_euler` orders and both order-specific
Euler readbacks over 67 shared inputs, including float32 neighbors around
positive and negative 90 degrees, fixed-seed cases, and values outside the
canonical range. It also records ten valid matrix reconstruction cases and
three native-only near-collinear basis diagnostics, axis-angle, from-to/
shortest-arc behavior, rotation-vector `exp`/`log`, shortest and
no-shortest-path SLERP, normalization edges, vector action, composition, and
an explicit source-frame map. The direct `Quaternion(Basis)` calls on the
diagnostic inputs intentionally emit Godot's orthonormality error and record
the returned identity result; those inputs are not passed to QuatOps.

The fixture's `canonical_interop` section adds direct native basis and
mixed-vector actions for the shared WXYZ inputs and a noncommuting product.

Run the separate QuatOps/SciPy comparisons with:

```powershell
python -m pytest tests\test_godot_native_fixtures.py tests\test_native_engine_extended_conversions.py tests\test_engine_proof_profiles.py tests\test_canonical_interoperability.py
```

This fixture covers Godot 4.7 stable only. It does not validate imported
assets, nonuniform-scale shear, or a double-precision build.
