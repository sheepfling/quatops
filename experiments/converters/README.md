# Auxiliary converter experiments

These scripts and fixtures are evidence tools, not QuatOps runtime code. Each
target has a versioned directory containing explicit-frame input generation
and the smallest native capture needed for its converter boundary. Captured
native outputs, contracts, and selected witnesses live under the matching
versioned path in `tests/golden/`; replay tests live in `tests/`.

## Layout

```text
experiments/converters/<target>/<target-version>/{generator,capture}
tests/golden/{oracles,transports}/<target>/<target-version>/{contract,witnesses,manifest}
tests/test_<target>_converter*.py
```

## Target runbooks

- [CesiumJS 1.124.0](cesium/1.124.0/README.md)
- [Eigen 5.0.1](eigen/5.0.1/README.md)
- [glTF 2.0.0 / Three.js 0.187.0](gltf/2.0.0/README.md)
- [MuJoCo 3.17.0](mujoco/3.17.0/README.md)
- [OpenCV 4.13.0](opencv/4.13.0/README.md)
- [ROS 2 Jazzy / tf2 0.36.23](ros2/0.36.23/README.md)
- [SpiceyPy 8.3.0](spiceypy/8.3.0/README.md)
- [OpenUSD 26.11](openusd/26.11/README.md)
- [PROJ 9.5.1 supporting capture](../oracles/proj/9.5.1/README.md)

Transport evidence may include a loader version below the format version,
such as glTF 2.0.0 loaded by Three.js 0.187.0. The source frame, destination
frame, active/passive meaning, algebra, coefficient order, and inverse rule
must be explicit in each fixture contract. Do not derive an asset's model axes
from an engine or format name.

## Captured converter proofs

| Target/version | Native boundary | Replay test | Scope |
| --- | --- | --- | --- |
| CesiumJS 1.124.0 | Local ENU orientation and WGS84 ENU-to-ECEF placement | `tests/test_cesium_converter.py` | Selected origins, points, directions, inverse recovery |
| glTF 2.0.0 / Three.js 0.187.0 | QuatOps-generated `.gltf` loaded through native `GLTFLoader` | `tests/test_gltf_converter_roundtrip.py` | Static parent/child hierarchy |
| OpenUSD 26.11 | QuatOps-generated USDA reopened and evaluated by OpenUSD | `tests/test_openusd_converter_roundtrip.py` | Orient/translate stack, parent/child and reset behavior |
| ROS 2 Jazzy / tf2 0.36.23 | Native `BufferCore` chain and `TransformStamped` message APIs | `tests/test_ros_tf2_converter.py` | Static named-frame chain and both lookup directions |
| SpiceyPy 8.3.0 / CSPICE_N0067 | Native `q2m`/`mxv`, `m2q`, `qxq`, and the `axisar` matrix/`m2q` route | `tests/test_spiceypy_converter.py` | Direct WXYZ quaternion API in the explicitly shared probe frame; excludes `eul2m` and named SPICE frames |
| MuJoCo 3.17.0 | Native `mju_quat2Mat`, `mju_rotVecQuat`, `mju_mat2Quat`, `mju_mulQuat`, and `mju_axisAngle2Quat` | `tests/test_mujoco_converter.py` | Direct WXYZ quaternion API in the explicitly shared probe frame; excludes model and simulator frame assignments |
| OpenCV 4.13.0 | Native `cv2.Rodrigues` and `cv2.gemm` | `tests/test_opencv_converter.py` | Rodrigues-vector round trip and matrix action in float32 and float64; excludes C++ `cv::Quat` |
| Eigen 5.0.1 | Native C++ `Quaternionf`, `Quaterniond`, `AngleAxis`, and matrix constructors | `tests/test_eigen_converter.py` | WXYZ constructor, XYZW coeffs, vector action, matrix round trip, and product order |

Each native exporter, dependency pin, and runbook is kept beside its versioned
target package. Shared inputs live in `experiments/oracles/shared/`; exact
selected native outputs live under `tests/golden/`.

To add a target version, add a new version directory and fixture directory;
keep prior native manifests and witness values immutable. Broad discovery
sweeps stay outside the retained witness files. A minimized native counterexample
belongs in a target regression fixture with the behavior it protects stated.
