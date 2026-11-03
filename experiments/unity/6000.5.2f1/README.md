# Unity Editor 6000.5.2f1 capture

This project captures the pinned `UnityEngine` quaternion API on WindowsEditor.
Its full discovery output is temporary. Selected native converter witnesses
are retained under [`tests/oracles/unity/6000.5.2f1`](../../../tests/oracles/unity/6000.5.2f1/contract.json).

From the repository root in PowerShell:

```powershell
$root = (Resolve-Path .).Path
$unity = 'C:\Program Files\Unity\Hub\Editor\6000.5.2f1\Editor\Unity.exe'
$project = Join-Path $root 'experiments\unity\6000.5.2f1'
$output = Join-Path $env:TEMP 'quatops-unity-full.json'
$log = Join-Path $env:TEMP 'quatops-unity-capture.log'
$arguments = "-batchmode -quit -projectPath `"$project`" -executeMethod QuatOpsEngineCapture.Capture -QuatOpsCaptureOutput=`"$output`" -logFile `"$log`""
$capture = Start-Process -FilePath $unity -ArgumentList $arguments -PassThru -Wait -WindowStyle Hidden
if ($capture.ExitCode -ne 0) { throw "Unity capture failed with exit code $($capture.ExitCode). See $log" }
.venv\Scripts\python.exe scripts\retain_engine_witnesses.py unity 6000.5.2f1 `
  $output tests\oracles\unity\6000.5.2f1
```

The retained set covers eight canonical orientations, native quaternion
coefficients and native basis/mixed-vector actions, and both orders of one
noncommuting product. The contract records API, component order, frame map,
precision, build, and capture provenance. Euler, animation, and unrelated API
sweeps remain discovery work rather than permanent converter goldens.

Run the replay and generated frame-map checks with:

```powershell
.venv\Scripts\python.exe -m pytest -q tests\test_canonical_interoperability.py tests\test_engine_basis_oracle.py tests\test_engine_proof_profiles.py
```
