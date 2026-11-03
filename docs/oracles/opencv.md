# OpenCV 4.13.0 Rodrigues oracle

The Wave 2 capture ran OpenCV 4.13.0 through Python 3.13.5 on Linux x64, with
NumPy 2.3.5. The native build information, pinned exporter, raw observations,
and their manifests are kept in the [capture packet](../../experiments/quatops_capture_wave2/README.md)
and copied into [the versioned golden folder](../../tests/golden/oracles/opencv/4.13.0/manifest.json).

The run contains 264 successful observations from `cv2.Rodrigues`: binary32
and binary64 rotation-vector/matrix conversions plus six forward Jacobian
cases. Six selected invalid-matrix probes are explicitly `not_run`, not
failures or successful calls. The official API documents the vector/matrix
and Jacobian route in [OpenCV 4.13 calibration documentation](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html).

Forward matrices agree with separate SciPy calculations to maximum element
errors of `2.98e-8` for binary32 and `4.45e-16` for binary64. The best
finite-difference Jacobian check is at most `2.64e-11` for the recorded cases.
The native matrix-to-vector round trip retains 26 binary64 near-zero/near-pi
diagnostics above `1e-12`, with a maximum element residual near `1e-6`.

This is Rodrigues evidence only. It does not capture OpenCV's C++ quaternion
algebra, and the inverse-boundary measurements are not promoted as behavior
that QuatOps should imitate.
