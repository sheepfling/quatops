# Unity Editor 6000.5.2f1 capture

This project captures `UnityEngine` and `Unity.Mathematics` on WindowsEditor.
`ProjectSettings/ProjectVersion.txt` records revision `eb73d3b415a1`; the
manifest declares `com.unity.mathematics` 1.4.0, resolved as a built-in module.
Values use float32 precision.

From the repository root in PowerShell, regenerate the native fixture with the
same Editor version:

```powershell
$root = (Resolve-Path .).Path
$unity = 'C:\Program Files\Unity\Hub\Editor\6000.5.2f1\Editor\Unity.exe'
$project = Join-Path $root 'experiments\unity\6000.5.2f1'
$output = Join-Path $root 'tests\golden\unity\6000.5.2f1\native_rotation.json'
$log = Join-Path $project 'capture.log'
$arguments = "-batchmode -quit -projectPath `"$project`" -executeMethod QuatOpsEngineCapture.Capture -QuatOpsCaptureOutput=`"$output`" -logFile `"$log`""
$capture = Start-Process -FilePath $unity -ArgumentList $arguments -PassThru -Wait -WindowStyle Hidden
if ($capture.ExitCode -ne 0) { throw "Unity capture failed with exit code $($capture.ExitCode). See $log" }
```

The capture records fifteen UnityEngine Euler cases, including four samples
around the ZXY middle-axis singularity. Its extended section captures all six
Unity Mathematics Euler constructors and inverse readbacks for 37 shared
triples per order (222 order/input pairs): 29 fixed mixed/boundary cases and
eight pseudo-random triples generated with `System.Random` seed `0x514F5053`.
The fixed inputs include a `(10, 20, 30)` sequence diagnostic,
near-singularity probes for each axis, and angles outside canonical ranges. It
also captures `Matrix4x4.rotation` and the Mathematics matrix
constructor; axis-angle and derived rotation-vector conversions; shortest-path
`Slerp`, clamped/unclamped `Lerp` and `math.nlerp`, normalization edges; basis
action, cross-products, quaternion/Transform composition, and an explicit
source-frame map. Its `canonical_interop` section captures the shared canonical
WXYZ inputs, native basis and mixed-vector actions, and a noncommuting product
for CAP-300. The selected public APIs do not expose a full-path SLERP
option. Unity's native normalization behavior is kept separate by API: zero
normalizes to identity in `UnityEngine`, while
`Unity.Mathematics.math.normalize` produces non-finite components for zero.
Those string-encoded components keep the capture valid JSON and are paired
with a finite flag.

Run the separate QuatOps/SciPy comparisons with:

```powershell
python -m pytest tests\test_unity_native_fixtures.py tests\test_native_engine_extended_conversions.py tests\test_canonical_interoperability.py
```

This fixture covers Unity 6000.5.2f1 only. It does not cover imported assets or
nonuniform-scale shear.
