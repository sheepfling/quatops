# Blender 5.1.2 capture

This version directory pins Blender 5.1.2, build hash `ec6e62d40fa9`, on
Windows with embedded Python 3.13.9. The script emits a full discovery capture
to a temporary file; only selected quaternion-converter witnesses are retained
in [`tests/oracles/blender/5.1.2`](../../../tests/oracles/blender/5.1.2/contract.json).

```powershell
$full = Join-Path $env:TEMP 'quatops-blender-full.json'
& 'C:\Program Files\Blender Foundation\Blender 5.1\blender.exe' `
  --background --factory-startup --python experiments\blender\5.1.2\capture.py `
  -- --output $full
if ($LASTEXITCODE -ne 0) { throw "Blender capture failed: $LASTEXITCODE" }
.venv\Scripts\python.exe scripts\retain_engine_witnesses.py blender 5.1.2 `
  $full tests\oracles\blender\5.1.2
```

The retained record contains eight canonical inputs, exact native WXYZ
quaternions, basis and mixed-vector actions, both orders of a noncommuting
product, the assigned identity probe frame, and capture/build provenance.
Euler, trajectory, and scene probes remain temporary until those adapter
features are in scope.
