# Godot 4.8-dev1 capture

This headless project captures the installed official Godot 4.8-dev1 build
(`ebbf577a041d1ab8d7824b6f90e7d0575461ca45`). Full discovery output is
temporary; selected converter witnesses are retained in
[`tests/oracles/godot/4.8-dev1`](../../../tests/oracles/godot/4.8-dev1/contract.json).

```powershell
$root = (Resolve-Path .).Path
$godot = 'C:\Users\Public\Godot\engines\windows\Godot_v4.8-dev1_win64.exe\Godot_v4.8-dev1_win64_console.exe'
$project = Join-Path $root 'experiments\godot\4.8-dev1'
$script = Join-Path $root 'experiments\godot\capture.gd'
$output = Join-Path $env:TEMP 'quatops-godot48-full.json'
& $godot --headless --path $project --script $script -- $output
if ($LASTEXITCODE -ne 0) { throw "Godot capture failed: $LASTEXITCODE" }
.venv\Scripts\python.exe scripts\retain_engine_witnesses.py godot 4.8-dev1 `
  $output tests\oracles\godot\4.8-dev1
```

The selected output retains eight canonical orientations, the exact native
quaternions, native basis/mixed-vector actions, both noncommuting product
orders, and the engine build record. The remaining Euler, matrix, and
interpolation sweeps are temporary discovery results.
