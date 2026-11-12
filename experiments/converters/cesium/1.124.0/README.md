# CesiumJS 1.124.0 local-ENU to ECEF converter capture

This capture executes generated `converter_inputs.json` against native CesiumJS
1.124.0 APIs. The [distribution pin](distribution.json) identifies the upstream
archive and its SHA-512 integrity digest. [`geographic_inputs_v1.json`](../../../oracles/shared/geographic_inputs_v1.json)
retains the selected WGS84 origin cases; canonical rotations come from the shared
[`canonical_inputs.json`](../../../interoperability/v1/canonical_inputs.json).

From the repository root, run the cross-platform helper. It installs the
verified standalone Cesium bundle, generates the inputs, and captures the selected
witnesses and provenance into the requested directory. The generated input is checked
against the captured input digest during evidence replay:

```shell
uv run --extra dev python -m experiments.run_converter_capture cesium --repository-root . --output <empty-capture-directory>
```

The helper requires Node.js 24 on `PATH` and network access to download the
pinned archive. It extracts only the package metadata and bundled Node entry
point into a temporary directory; npm and the Cesium widget dependencies are
not needed. The original capture's npm manifest and lock remain in the
historical provenance archive.
The capture accepts finite binary64 values and preserves signed zero in JSON.
Omit `--output` to create a persistent temporary capture directory. The
historical fixture is not overwritten.

The selected native values are retained in
[`tests/golden/transports/cesium/1.124.0`](../../../../tests/golden/transports/cesium/1.124.0/manifest.json)
and replayed by
[`tests/experiments/converters/cesium/test_cesium_converter.py`](../../../../tests/experiments/converters/cesium/test_cesium_converter.py).
