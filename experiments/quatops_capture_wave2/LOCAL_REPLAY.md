# Wave 2 import and local follow-on

## Received packet

The supplied `quatops_capture_wave2.zip` is retained as
[`upstream_wave2.zip`](upstream_wave2.zip), SHA-256
`7bc6eb3384ae3dc1eb2b326abb2abb07adc0321e7a831de2565bee0e554fe5bf`. Before
local changes, all 77 files listed by its `SHA256SUMS` matched. The imported
Linux runs and their manifests remain unchanged under `runs/`.

The source packet's four native library runs are still Linux evidence: OpenCV
4.13.0, VTK 9.6.2, Trimesh 4.11.1, and PROJ 9.5.1/pyproj 3.7.2 under Python
3.13.5. Their observations were copied byte-for-byte to the matching
versioned paths under `tests/golden/`. Windows replay checked their hashes,
schemas, binary encodings, and separate SciPy/reference comparisons; it did not
rerun those four native libraries.

## Local native captures

Using Node.js `v24.21.0` on Windows x64, this workspace executed:

- CesiumJS package `1.124.0`: 171 observations across local HPR, matrix
  conversion, Hamilton multiplication, SLERP, and WGS84 geographic placement.
- Three.js `0.187.0` GLTFLoader: 649 observations, including 534 Euler cases,
  three multiplication cases, twelve static assets, and 100 animation samples.
- Khronos glTF Validator `2.0.0-dev.3.10`: twelve assets, zero errors,
  warnings, infos, or hints.

These outputs are stored in versioned `tests/golden/` folders. The
[derived reference report](verification/local_native_comparison.json) uses
SciPy `1.17.0` and an independent glTF evaluator; its tests are separate from
the Node runtime capture. It explicitly records that QuatOps was not run.

The supplied package's `capture_io.mjs` needed a Windows `cmd.exe` route to
record `npm ls`. Cesium's unconstrained dependency ranges initially resolved a
newer `@zip.js/zip.js` that broke Cesium 1.124's `zip-no-worker.js` import. The
local `package.json` and lock therefore pin the compatible Cesium engine,
widgets, and zip.js versions; the exact graph is retained. The first failed
Three.js attempt left observations but no manifest, so that file is kept under
`verification/first-attempt-three-output/` and is not treated as a capture.

## Replay

The local replay environment is Python `3.13.14`, NumPy `2.3.5`, SciPy
`1.17.0`, pytest `9.0.2`, and jsonschema `4.26.0`.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m tools.verify_local_native
node .\js\test_capture_io.mjs
```

The wave package suite passes 184 tests. The first Windows replay exposed
POSIX-path assumptions and a last-bit PROJ/libm difference in one exact-dict
comparison; the test helper now relocates recorded paths independent of the
host path separator and compares derived numeric reports with an absolute
`1e-7` tolerance. Raw native values and hashes are still checked exactly.

## ROS/tf2 follow-on and remaining work

The packet's original ROS 2 source was not itself a native capture. A separate
local follow-on now builds and runs a C++20 exporter in the pinned ROS 2 Jazzy
container. Its selected native `BufferCore` frame-chain observations and
QuatOps replay are documented in [the ROS/tf2 proof](../../docs/oracles/ros2-tf2.md).
OpenXR remains planned; it has no pinned runtime capture in this workspace.
