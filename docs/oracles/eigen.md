# Eigen 5.0.1 quaternion boundary oracle

The retained native capture was compiled against the official Eigen 5.0.1
source archive with GCC 15.2.0 on Windows x86-64 using C++17. The source
archive URL, SHA-256, and the hash of `Eigen/Version` are in
[`source.json`](../../experiments/oracles/eigen/5.0.1/source.json). The native
C++ harness and its selected output manifest live under
[`tests/golden/oracles/eigen/5.0.1/converter`](../../tests/golden/oracles/eigen/5.0.1/converter/manifest.json).

## Measured behavior

For unit Hamilton WXYZ inputs in the explicitly assigned probe frame, the
native `Eigen::Quaternion(w, x, y, z)` accessors preserve WXYZ and
`coeffs()` exposes XYZW. `Quaternion * Vector3` matched the active Hamilton
action for three basis vectors and an asymmetric probe. `toRotationMatrix()`
matched the same action; its native column-major coefficient storage is retained
separately from the logical row display. Constructing a quaternion from that
matrix reproduced the orientation. `lhs * rhs` matched Hamilton product order,
with the right operand acting first. A native `AngleAxis` constructor path was
also captured. Both `Quaternionf` and `Quaterniond` are covered.

The native C++ capture does not import QuatOps. Replay tests compare its frozen
outputs with QuatOps and a separate SciPy matrix calculation in
[`tests/test_eigen_converter.py`](../../tests/test_eigen_converter.py).
Eigen is a mathematical library here; the assigned axes do not claim any
application-specific world, body, camera, or asset frame.

The official [Eigen quaternion API reference](https://eigen.tuxfamily.org/dox/classEigen_1_1Quaternion.html)
documents the WXYZ constructor parameter order and XYZW `coeffs()` storage.
