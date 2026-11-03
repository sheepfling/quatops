# QuatOps orientation captures — Wave 2

Prepared November 2, 2026 (America/Chicago); native run manifests retain their actual UTC timestamps.

As delivered, this bundle contains **399 successful library observations**, six explicitly unrun invalid-input probes, and separate mathematical comparisons. The later workspace captures, replay fixes, and their results are recorded in [LOCAL_REPLAY.md](LOCAL_REPLAY.md); the packet's original `runs/` data remains unchanged.

## What actually ran

| Runtime | Observations | Scope |
|---|---:|---|
| OpenCV 4.13.0 | 264 successful; 6 intentionally not run | Rodrigues vector/matrix conversion, both precisions, native Jacobians and inverse reconstruction |
| VTK 9.6.2 | 112 successful | 12 static glTF/GLB imports plus 100 native animation samples |
| Trimesh 4.11.1 | 12 successful | Static scene-graph import of the same assets; bundled JSON-schema checks |
| PROJ 9.5.1 / pyproj 3.7.2 | 11 successful | WGS84 geodetic/ECEF and topocentric ENU probes supporting the upcoming Cesium capture |

The glTF corpus has **six scenarios**, each encoded as both `.gltf` and `.glb`: static hierarchy, STEP, LINEAR, CUBICSPLINE, globally negated cubic keys/tangents, and one locally negated cubic key/tangents. Encoding the same case twice does not create two independent scenarios. Animation times are nonuniform and cubic tangents are nonzero.

In the supplied packet, **ROS/tf2, Cesium, OpenUSD, Three.js, SpiceyPy/CSPICE, and MuJoCo** were not run in its delivery environment. A later local follow-on ran Cesium, Three.js, and the Khronos validator; see [LOCAL_REPLAY.md](LOCAL_REPLAY.md). ROS/tf2, SpiceyPy/CSPICE, and MuJoCo already have separate QuatOps captures; OpenUSD also has a separate transport capture. The packet's own OpenUSD, ROS, SpiceyPy, and MuJoCo exporters remain unrun. The local npm lock and Cesium dependency pins were added after delivery.

## Delivered work

- `exporters/`: native-library capture scripts, including unrun OpenUSD, ROS wrapper, CSPICE and MuJoCo exporters.
- `ros2/quatops_ros_capture/`: C++ tf2/BufferCore capture package, CMake build and package metadata.
- `js/`: Cesium math/geographic capture, Three.js math/GLTFLoader capture, and Khronos validator driver.
- `assets/`: 12 glTF/GLB files, two text-authored USD stages, sampling times.
- `corpus/`: original frozen shared corpus plus 11 geographic locations and three HPR probes.
- `runs/`: only actual library-run observations and manifests. No fake ROS/Cesium/USD goldens.
- `verification/`: comparison report, runtime inventory, logs and pytest JUnit output.
- `tests/`: schema, bit serialization, integrity, format, reference-comparison and independence checks.
- `docs/`: capture ledger, runnable instructions, findings, integration notes and primary sources.

## Results

All **124 glTF consumer observations** agree with separate TRS/animation reference evaluation at a `1e-6` maximum-matrix-element diagnostic threshold. Maximum errors: VTK `2.399441831335558e-7`; Trimesh `8.881784197001252e-16`. These are transform-element differences, not angular errors.

OpenCV forward matrix differences versus SciPy: binary32 `2.979167001981864e-8`; binary64 `4.440892098500626e-16`. Its own inverse round trips expose **26 binary64 boundary discrepancies** above `1e-12`, with a maximum matrix-element residual about `1e-6`. They remain recorded separately. Six forward Jacobians were checked at three finite-difference steps; the largest best-step error was `2.6320778889754592e-11`.

PROJ analytic point comparisons have maximum absolute error `9.313225746154785e-10 m`; native local-coordinate round-trip residual is at most `6.343396918850885e-10 m`. These are numerical consistency measurements, not real-world geodetic accuracy or a Cesium execution.

The supplied packet reported **179 pytest tests passed** in its original environment. The local Windows replay, after portability fixes and the Cesium/Three.js checks, passes **184 tests**; these mix structural, replay and numerical checks and are not 184 independent native cases. JavaScript numeric serialization checks and source syntax checks also passed. ROS/tf2 and OpenXR remain unrun.

## Replay

From this bundle's root in the pinned Python environment:

```bash
python -m tools.verify_results
python -m pytest -q
node js/test_capture_io.mjs
```

Use [RUNBOOK.md](docs/RUNBOOK.md) for native capture commands. Never overwrite the supplied runs. Captures start as `captured_not_reviewed`; independent review and QuatOps integration are separate gates.

[Capture ledger](docs/CAPTURE_LOG.md) · [Findings and limitations](docs/FINDINGS.md) · [Integration checklist](docs/INTEGRATION.md) · [Sources](docs/SOURCES.md)
