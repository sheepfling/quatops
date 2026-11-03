# Blender 5.1.2 capture

This version directory pins Blender 5.1.2, build hash `ec6e62d40fa9`, on
Windows with embedded Python 3.13.9. The native fixture is
[`native_rotation.json`](../../../tests/golden/blender/5.1.2/native_rotation.json).

From the repository root, run the native capture with:

```powershell
& 'C:\Program Files\Blender Foundation\Blender 5.1\blender.exe' `
  --background --factory-startup --python experiments\blender\5.1.2\capture.py
```

The script writes to the matching versioned fixture path. Pass an alternate
output path after Blender's `--` separator when reviewing a capture:

```powershell
& 'C:\Program Files\Blender Foundation\Blender 5.1\blender.exe' `
  --background --factory-startup --python experiments\blender\5.1.2\capture.py `
  -- --output C:\Temp\blender-native-review.json
```

Then compare the frozen native values against the 23 QuatOps representations
and SciPy with:

```powershell
.venv\Scripts\python.exe -m pytest -q tests\test_blender_native_fixtures.py
```

The `canonical_interop` section also captures native WXYZ basis and mixed-vector
actions for the common CAP-300 inputs in the explicitly assigned probe frame.
The first command launches Blender. The pytest command reads only the frozen
JSON and runs after the engine has exited; add
`tests\test_canonical_interoperability.py` to replay the converter proof.
