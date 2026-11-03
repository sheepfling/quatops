# Godot 4.7 stable capture

This headless project captures the installed official Godot 4.7 stable build.
The shared exporter records only the selected quaternion converter witnesses,
retained in [`tests/oracles/godot/4.7-stable`](../../../tests/oracles/godot/4.7-stable/contract.json).

```powershell
$root = (Resolve-Path .).Path
$godot = 'C:\Users\Public\Godot\engines\windows\Godot_v4.7-stable_win64.exe\Godot_v4.7-stable_win64_console.exe'
$project = Join-Path $root 'experiments\godot\4.7-stable'
$script = Join-Path $root 'experiments\godot\capture.gd'
$output = Join-Path $env:TEMP 'quatops-godot47-witnesses.json'
& $godot --headless --path $project --script $script -- $output
if ($LASTEXITCODE -ne 0) { throw "Godot capture failed: $LASTEXITCODE" }
.venv\Scripts\python.exe scripts\retain_engine_witnesses.py godot 4.7-stable `
  $output tests\oracles\godot\4.7-stable
```

The output contains eight canonical orientations, exact native quaternions,
native basis/mixed-vector actions, both noncommuting product orders, and the
engine build record.
