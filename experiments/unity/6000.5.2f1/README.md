# Unity Editor 6000.5.2f1 capture

This project captures selected outputs from the pinned `UnityEngine`
quaternion API on WindowsEditor; it has no external Unity package dependency.
The temporary output is retained under
[`tests/oracles/unity/6000.5.2f1`](../../../tests/oracles/unity/6000.5.2f1/contract.json).
Only `ProjectVersion.txt` and the package manifest are retained; Unity
regenerates editor and player defaults when the project is opened.

From the repository root in PowerShell:

```powershell
$root = (Resolve-Path .).Path
$unity = 'C:\Program Files\Unity\Hub\Editor\6000.5.2f1\Editor\Unity.exe'
$project = Join-Path $root 'experiments\unity\6000.5.2f1'
$output = Join-Path $env:TEMP 'quatops-unity-witnesses.json'
$log = Join-Path $env:TEMP 'quatops-unity-capture.log'
$arguments = @(
  '-batchmode', '-quit', '-projectPath', "`"$project`"",
  '-executeMethod', 'QuatOpsEngineCapture.Capture',
  "-QuatOpsCaptureOutput=`"$output`"", '-logFile', "`"$log`""
)
$capture = Start-Process -FilePath $unity -ArgumentList $arguments -PassThru -Wait -WindowStyle Hidden
if ($capture.ExitCode -ne 0) { throw "Unity capture failed with exit code $($capture.ExitCode). See $log" }
.venv\Scripts\python.exe -m scripts.retain_engine_witnesses unity 6000.5.2f1 `
  $output "$env:TEMP\quatops-engine-review\unity\6000.5.2f1" --repository-root .
```

The capture emits five canonical orientations, native quaternion
coefficients and native basis/mixed-vector actions, and both orders of one
noncommuting product. The contract records API, component order, frame map,
precision, build, and capture provenance. Euler and animation adapters are
outside this converter's retained evidence.

Run the replay and generated frame-map checks with:

```shell
python -m scripts.verify engines
```
