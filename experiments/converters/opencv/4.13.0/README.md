# OpenCV 4.13.0 Rodrigues converter-boundary capture

This auxiliary proof covers the OpenCV rotation-vector boundary because
`cv2.Rodrigues` consumes/returns Rodrigues vectors and matrices rather than
quaternions. `prepare_inputs.py` uses QuatOps to prepare rotation vectors from
the shared canonical WXYZ inputs. `capture_native.py` imports no QuatOps code;
it calls the pinned native `cv2.Rodrigues` and `cv2.gemm` APIs for float32 and
float64 profiles.

The captured runtime used CPython 3.13.14, NumPy 2.3.5, and
`opencv-python-headless==4.13.0.92`. The versioned project installs QuatOps,
the shared experiment models, and this target's optional dependency group in
one locked environment. From the repository root, run:

```shell
uv run --extra dev python -m experiments.run_converter_capture opencv --repository-root .
```

The retained manifest records the OpenCV package version, runtime details, and
native binary hashes alongside the selected witnesses. The verbose build
configuration and generated QuatOps input file are not retained; the shared
source plan and exact native-observed rotation vectors remain available.
Replay the selected boundary evidence with
[`test_opencv_converter.py`](../../../../tests/experiments/converters/opencv/test_opencv_converter.py).
