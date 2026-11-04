# Shared native-capture inputs and helpers

[`orientation_inputs_v1.json`](orientation_inputs_v1.json) is a shared input
corpus for mathematical oracle tests and selected ROS 2/tf2 cases. The broader
discovery outputs are not retained per target; converter packages refer to
the smaller selected set under `experiments/interoperability/v1/`.

[`capture_io.py`](capture_io.py) provides a small standard-library capture
envelope for native Python wrappers. It records input, output, source, runtime,
precision, and binary identities without implementing rotation formulas or
expected values.

[`geographic_inputs_v1.json`](geographic_inputs_v1.json) contains finite probes,
not a certification that every coordinate is physically valid WGS84. Wrapped
longitudes, exact poles, and latitude values outside `[-90, 90]` are retained
as target probes. Cesium 1.124 evaluates its geodetic trigonometric expressions;
PROJ 9.5.1 rejects out-of-range latitude with `errcheck=True`, which aborts its
current capture instead of recording a native error observation. See the
[Cesium source](https://github.com/CesiumGS/cesium/blob/1.124/packages/engine/Source/Core/Cartesian3.js#L876-L948)
and [PROJ source](https://github.com/OSGeo/PROJ/blob/9.5.1/src/fwd.cpp#L58-L77)
for those target-specific behaviors.
