# Blender 5.1.2 capture

This version directory pins Blender 5.1.2, build hash `ec6e62d40fa9`, on
Windows with embedded Python 3.13.9. The script emits selected
quaternion-converter witnesses to a temporary file, retained in
[`tests/oracles/blender/5.1.2`](../../../tests/oracles/blender/5.1.2/contract.json).

```powershell
$witnesses = Join-Path $env:TEMP 'quatops-blender-witnesses.json'
& 'C:\Program Files\Blender Foundation\Blender 5.1\blender.exe' `
  --background --factory-startup --python experiments\blender\5.1.2\capture.py `
  -- --output $witnesses
if ($LASTEXITCODE -ne 0) { throw "Blender capture failed: $LASTEXITCODE" }
.venv\Scripts\python.exe -m scripts.retain_engine_witnesses blender 5.1.2 `
  $witnesses tests\oracles\blender\5.1.2 --repository-root .
```

The retained record contains five canonical inputs, exact native WXYZ
quaternions, basis and mixed-vector actions, both orders of a noncommuting
product, the assigned identity probe frame, and capture/build provenance.
