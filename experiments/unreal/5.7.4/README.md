# Unreal Engine 5.7.4 capture

`capture_python.py` is a supplemental Unreal Editor Python probe. It records
the Python-exposed `FRotator`/`FQuat` values without importing QuatOps or SciPy.
The canonical fixture comes from the C++ Editor Automation Test, which captures
`FRotator3f/FQuat4f` and `FRotator3d/FQuat4d` directly.

The native build reports `5.7.4-51494982+++UE5+Release-5.7` on WindowsEditor
x64. From the repository root in PowerShell, run the capture with the selected
Unreal installation:

```powershell
$engine = 'C:\Program Files\Epic Games\UE_5.7'
$project = "$PWD\experiments\unreal\5.7.4\QuatOpsUnrealCapture.uproject"
& "$engine\Engine\Binaries\Win64\UnrealEditor-Cmd.exe" $project `
  -run=pythonscript `
  -script="$PWD\experiments\unreal\5.7.4\capture_python.py" `
  -unattended -nop4 -nosplash -nullrhi
```

That Python probe writes its supplemental output under the project's ignored
`Saved/` folder. The canonical native fixture is produced by the compiled C++
Automation Test:

```powershell
$engine = 'C:\Program Files\Epic Games\UE_5.7'
$project = "$PWD\experiments\unreal\5.7.4\QuatOpsUnrealCapture.uproject"
& "$engine\Engine\Build\BatchFiles\Build.bat" `
  QuatOpsUnrealCaptureEditor Win64 Development "-Project=$project" -WaitMutex
& "$engine\Engine\Binaries\Win64\UnrealEditor-Cmd.exe" $project `
  '-ExecCmds=Automation RunTests QuatOps.Unreal.CaptureGoldenFixtures; Quit' `
  -unattended -nop4 -nosplash -nullrhi `
  "-QuatOpsCaptureOutput=$PWD\tests\golden\unreal\5.7.4\native_rotation.json"
if ($LASTEXITCODE -ne 0) { throw "Unreal automation capture failed with exit code $LASTEXITCODE" }
```

The JSON contains fifteen Euler/Rotator cases, matrix conversion and
axis-angle/rotation-vector constructors/readbacks, shortest and full-path SLERP
samples, normalization edges, cross products, quaternion/transform composition,
and a source map from the explicitly declared right-handed frame (+X right,
+Y forward, +Z up). It captures API differences for float32 and float64 and
native-only diagnostics for malformed matrices and singular interpolation.
Its `canonical_interop` section also captures float32 and float64 target
quaternion coefficients, all three native basis actions, a mixed-vector action,
and a noncommuting native product for the shared WXYZ inputs in
[`experiments/interoperability/v1`](../../interoperability/README.md).

Run the separate QuatOps/SciPy checks with:

```powershell
python -m pytest tests\test_unreal_engine_golden.py tests\test_unreal_native_semantics.py tests\test_unreal_extended_conversions.py tests\test_canonical_interoperability.py
```

The engine test does not validate imported assets, a universal model-forward
axis, or nonuniform-scale shear.
