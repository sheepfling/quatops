# CesiumJS 1.124.0 local-ENU to ECEF converter capture

This capture executes the QuatOps-generated inputs in
[`converter_inputs.json`](converter_inputs.json) against native CesiumJS
1.124.0 APIs. The local package manifest and lockfile pin Cesium independently
of other targets. [`geographic_inputs_v1.json`](../../../oracles/shared/geographic_inputs_v1.json)
retains the selected WGS84 origin cases; canonical rotations come from the shared
[`canonical_inputs.json`](../../../interoperability/v1/canonical_inputs.json).

From the repository root, run the cross-platform helper. It installs the
versioned Node dependencies, generates the inputs, captures in a temporary
directory, and retains the selected witnesses:

```shell
uv run --extra dev python -m experiments.run_converter_capture cesium --repository-root .
```

The helper requires Node.js and npm on `PATH`; the npm lockfile pins Cesium.

The selected native values are retained in
[`tests/golden/transports/cesium/1.124.0`](../../../../tests/golden/transports/cesium/1.124.0/manifest.json)
and replayed by
[`tests/experiments/converters/cesium/test_cesium_converter.py`](../../../../tests/experiments/converters/cesium/test_cesium_converter.py).
