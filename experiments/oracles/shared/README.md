# Shared native-capture inputs and helpers

Converter captures keep their inputs beside their target-specific exporters,
or use the shared canonical inputs under `experiments/interoperability/v1/`.

[`capture_io.py`](capture_io.py) provides a typed JSON capture envelope for
native Python wrappers. Target-specific capture models validate observations
before they enter this generic provenance container. It records inputs,
outputs, source, runtime, precision, and binary identities without implementing
rotation formulas or expected values.

JSON is the storage format, not the semantic type system. Generic envelope
values use Pydantic's recursive `JsonValue`; each experiment's raw observations
are validated against its target-specific strict schema before retention. The
experiment-owned `semantic_contract.json` records the target interpretation,
and the repository-side serializer/normalizer reads that contract after the
pinned runtime exits. Capture code does not import the common normalizer.

[`geographic_inputs_v1.json`](geographic_inputs_v1.json) contains finite probes,
not a certification that every coordinate is physically valid WGS84. Wrapped
longitudes, exact poles, and latitude values outside `[-90, 90]` are retained
as target probes. Cesium 1.124 evaluates its geodetic trigonometric expressions;
PROJ 9.5.1 rejects out-of-range latitude with `errcheck=True`, which aborts its
current capture instead of recording a native error observation. See the
[Cesium source](https://github.com/CesiumGS/cesium/blob/1.124/packages/engine/Source/Core/Cartesian3.js#L876-L948)
and [PROJ source](https://github.com/OSGeo/PROJ/blob/9.5.1/src/fwd.cpp#L58-L77)
for those target-specific behaviors.
