# Unreal 5.7.4 capture

This pinned Editor project captures selected quaternion converter witnesses
with its C++ Automation Test. The temporary output is retained in
[`tests/oracles/unreal/5.7.4`](../../../tests/oracles/unreal/5.7.4/contract.json).

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
  "-QuatOpsCaptureOutput=$env:TEMP\quatops-unreal-witnesses.json"
if ($LASTEXITCODE -ne 0) { throw "Unreal capture failed: $LASTEXITCODE" }
.venv\Scripts\python.exe -m scripts.retain_engine_witnesses unreal 5.7.4 `
  "$env:TEMP\quatops-unreal-witnesses.json" tests\oracles\unreal\5.7.4 --repository-root .
```

The capture uses `FQuat4f` and `FQuat4d` and emits eight explicit unit inputs,
their native quaternion values and native basis/mixed-vector actions, and both
orders of a noncommuting product. Native build, capture-source, input, and
capture hashes are recorded with the retained contract.

Replay the formula and frozen native-witness checks with:

```powershell
$engineProofs = @(
  'tests\quatops\experimental\engines\test_canonical_interoperability.py',
  'tests\quatops\experimental\engines\test_engine_basis_oracle.py',
  'tests\experiments\engines\shared\test_engine_proof_profiles.py'
)
.venv\Scripts\python.exe -m pytest -q $engineProofs
```
