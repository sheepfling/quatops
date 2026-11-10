# CesiumJS 1.124.0 local-ENU to ECEF converter capture

This capture executes generated `converter_inputs.json` against native CesiumJS
1.124.0 APIs. The local package manifest and lockfile pin Cesium independently
of other targets. [`geographic_inputs_v1.json`](../../../oracles/shared/geographic_inputs_v1.json)
retains the selected WGS84 origin cases; canonical rotations come from the shared
[`canonical_inputs.json`](../../../interoperability/v1/canonical_inputs.json).

From the repository root, run the cross-platform helper. It installs the
versioned Node dependencies, generates the inputs, and captures the selected
witnesses and provenance into the requested directory. The generated input is checked
against the captured input digest during evidence replay:

```shell
uv run --extra dev python -m experiments.run_converter_capture cesium --repository-root . --output <empty-capture-directory>
```

The helper requires Node.js 24 and npm on `PATH`; the npm lockfile pins Cesium.
The capture accepts finite binary64 values and preserves signed zero in JSON.
Omit `--output` to create a persistent temporary capture directory. The
historical fixture is not overwritten.

The selected native values are retained in
[`tests/golden/transports/cesium/1.124.0`](../../../../tests/golden/transports/cesium/1.124.0/manifest.json)
and replayed by
[`tests/experiments/converters/cesium/test_cesium_converter.py`](../../../../tests/experiments/converters/cesium/test_cesium_converter.py).
