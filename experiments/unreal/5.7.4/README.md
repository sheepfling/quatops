# Unreal 5.7.4 capture

This pinned Editor project produces a full discovery capture with its C++
Automation Test. The full output is temporary; only the selected CAP-300 native
witnesses are retained in [`tests/oracles/unreal/5.7.4`](../../../tests/oracles/unreal/5.7.4/contract.json).

From the repository root in PowerShell:

```powershell
$engine = 'C:\Program Files\Epic Games\UE_5.7'
$project = "$PWD\experiments\unreal\5.7.4\QuatOpsUnrealCapture.uproject"
& "$engine\Engine\Build\BatchFiles\Build.bat" `
  QuatOpsUnrealCaptureEditor Win64 Development "-Project=$project" -WaitMutex
if ($LASTEXITCODE -ne 0) { throw "Unreal build failed: $LASTEXITCODE" }
& "$engine\Engine\Binaries\Win64\UnrealEditor-Cmd.exe" $project `
  '-ExecCmds=Automation RunTests QuatOps.Unreal.CaptureGoldenFixtures; Quit' `
  -unattended -nop4 -nosplash -nullrhi `
  "-QuatOpsCaptureOutput=$env:TEMP\quatops-unreal-full.json"
if ($LASTEXITCODE -ne 0) { throw "Unreal capture failed: $LASTEXITCODE" }
.venv\Scripts\python.exe scripts\retain_engine_witnesses.py unreal 5.7.4 `
  "$env:TEMP\quatops-unreal-full.json" tests\oracles\unreal\5.7.4
```

The capture uses `FQuat4f` and `FQuat4d` and records the full API exploration
to the temporary output. The selector retains eight explicit unit inputs,
their native quaternion values and native basis/mixed-vector actions, and both
orders of a noncommuting product. Native build, capture-source, input, and
original-capture hashes are recorded with the retained contract.

Replay the formula and frozen native-witness checks with:

```powershell
.venv\Scripts\python.exe -m pytest -q tests\test_canonical_interoperability.py tests\test_engine_basis_oracle.py tests\test_engine_proof_profiles.py
```
