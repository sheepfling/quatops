# OpenCV 4.13.0 Rodrigues converter-boundary capture

This auxiliary proof covers the OpenCV rotation-vector boundary because
`cv2.Rodrigues` consumes/returns Rodrigues vectors and matrices rather than
quaternions. `prepare_inputs.py` uses QuatOps to prepare rotation vectors from
the shared canonical WXYZ inputs. `capture_native.py` imports no QuatOps code;
it calls the pinned native `cv2.Rodrigues` and `cv2.gemm` APIs for float32 and
float64 profiles.

The Windows capture used CPython 3.13.14, NumPy 2.3.5, and
`opencv-python-headless==4.13.0.92`. Create the external runtime environment
and execute from the repository root:

```powershell
$envDir = Join-Path $env:TEMP 'quatops-captures\opencv-4.13.0'
uv venv --python 3.13.14 $envDir
uv pip install --python (Join-Path $envDir 'Scripts\python.exe') -r experiments\converters\opencv\4.13.0\requirements-converter.txt
.venv\Scripts\python.exe experiments\converters\opencv\4.13.0\prepare_inputs.py
& (Join-Path $envDir 'Scripts\python.exe') experiments\converters\opencv\4.13.0\capture_native.py
```

The retained manifest records the OpenCV package version, runtime details, and
native binary hashes alongside the selected witnesses. The verbose build
configuration is not retained. Replay the selected boundary evidence with
`tests/experiments/converters/opencv/test_opencv_converter.py`.
