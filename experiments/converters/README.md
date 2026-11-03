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

The ROS exporter and pinned container recipe remain alongside the imported
Wave 2 capture source under `experiments/quatops_capture_wave2/ros2/`; its
selected evidence is versioned at `tests/golden/oracles/ros2/0.36.23/`.
This preserves the original capture-source grouping while keeping the accepted
fixture keyed by actual runtime version.

To add a target version, add a new version directory and fixture directory;
keep prior native manifests and witness values immutable. Broad discovery
sweeps stay outside the retained witness files. A minimized native counterexample
belongs in a target regression fixture with the behavior it protects stated.
