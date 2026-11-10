# OpenCV 4.13.0 Rodrigues converter boundary

OpenCV's `cv2.Rodrigues` API represents rotations as rotation vectors and
matrices, rather than exposing its quaternion type as the boundary used by this
adapter. QuatOps converts canonical active Hamilton WXYZ rotations to
Rodrigues vectors; the pinned OpenCV runtime converts those vectors to native
matrices, which are then used for basis and probe actions. The reverse route
converts native matrices back to rotation vectors and then to QuatOps
quaternions.

The Windows x64 capture used CPython `3.13.14`, NumPy `2.3.5`, and
`opencv-python-headless==4.13.0.92` (OpenCV `4.13.0`). It retains five
canonical orientation witnesses, a native-created axis-angle case, and both
orders of a noncommuting product, for float32 and float64. The native capture
uses `cv2.Rodrigues` and `cv2.gemm` and does not import QuatOps.

The selected native matrices, basis/probe images, and build identity are in
the [versioned converter fixture](../../tests/golden/oracles/opencv/4.13.0/converter/manifest.json).
The [capture project](../../experiments/converters/opencv/4.13.0/README.md)
contains the pinned requirements and input preparation/native capture tools.
Replay is covered by [test_opencv_converter.py](../../tests/experiments/converters/opencv/test_opencv_converter.py).
SciPy matrix comparisons during replay are formula checks, separate from the
captured OpenCV execution.

This accepts the Rodrigues vector/matrix boundary for the explicitly assigned
numeric probe frame. It does not validate C++ `cv::Quat` algebra,
camera/world extrinsics, Jacobians as a QuatOps adapter, or application-specific
frames. The [OpenCV 4.13 Rodrigues API](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html)
documents the native conversion used here.
