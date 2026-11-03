# Godot 4.7 stable capture

This headless project captures the installed official Godot 4.7 stable build.
Full discovery output is temporary; selected converter witnesses are retained
in [`tests/oracles/godot/4.7-stable`](../../../tests/oracles/godot/4.7-stable/contract.json).

```powershell
$root = (Resolve-Path .).Path
$godot = 'C:\Users\Public\Godot\engines\windows\Godot_v4.7-stable_win64.exe\Godot_v4.7-stable_win64_console.exe'
$project = Join-Path $root 'experiments\godot\4.7-stable'
$output = Join-Path $env:TEMP 'quatops-godot47-full.json'
& $godot --headless --path $project --script res://capture.gd -- $output
if ($LASTEXITCODE -ne 0) { throw "Godot capture failed: $LASTEXITCODE" }
.venv\Scripts\python.exe scripts\retain_engine_witnesses.py godot 4.7-stable `
  $output tests\oracles\godot\4.7-stable
```

The selected output retains eight canonical orientations, the exact native
quaternions, native basis/mixed-vector actions, both noncommuting product
orders, and the engine build record. The remaining Euler, matrix, and
interpolation sweeps are temporary discovery results.
