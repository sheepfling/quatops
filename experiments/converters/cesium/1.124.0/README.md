# CesiumJS 1.124.0 local-ENU to ECEF converter capture

This capture executes the QuatOps-generated inputs in
[`converter_inputs.json`](converter_inputs.json) against native CesiumJS
1.124.0 APIs. The local package manifest and lockfile pin Cesium independently
of other targets. [`geographic_inputs.json`](geographic_inputs.json) retains
the selected WGS84 origin cases; canonical rotations come from the shared
[`canonical_inputs.json`](../../../interoperability/v1/canonical_inputs.json).

From the repository root, install the pinned Node dependencies, generate the
inputs, and capture into a temporary directory:

```powershell
npm ci --prefix experiments\converters\cesium\1.124.0
uv run python experiments\converters\cesium\1.124.0\create_inputs.py
$captureDir = Join-Path $env:TEMP 'quatops-captures\cesium-1.124.0'
node experiments\converters\cesium\1.124.0\capture_native.mjs $captureDir
uv run python scripts\retain_cesium_converter_witnesses.py $captureDir
```

The selected native values are retained in
[`tests/golden/transports/cesium/1.124.0`](../../../../tests/golden/transports/cesium/1.124.0/manifest.json)
and replayed by
[`tests/experiments/converters/cesium/test_cesium_converter.py`](../../../../tests/experiments/converters/cesium/test_cesium_converter.py).
