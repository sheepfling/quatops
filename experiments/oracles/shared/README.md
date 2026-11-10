# Shared native-capture inputs and helpers

Converter captures keep their inputs beside their target-specific exporters,
or use the shared canonical inputs under `experiments/interoperability/v1/`.

Captures validate native observations with target-specific schemas and record
their provenance without implementing rotation formulas or expected values.

[`geographic_inputs_v1.json`](geographic_inputs_v1.json) contains finite probes,
not a certification that every coordinate is physically valid WGS84. Wrapped
longitudes, exact poles, and latitude values outside `[-90, 90]` are retained
as target probes. Cesium 1.124 evaluates its geodetic trigonometric expressions;
PROJ 9.5.1 rejects out-of-range latitude with `errcheck=True`, which aborts its
current capture instead of recording a native error observation. See the
[Cesium source](https://github.com/CesiumGS/cesium/blob/1.124/packages/engine/Source/Core/Cartesian3.js#L876-L948)
and [PROJ source](https://github.com/OSGeo/PROJ/blob/9.5.1/src/fwd.cpp#L58-L77)
for those target-specific behaviors.
