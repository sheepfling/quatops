# CesiumJS 1.124.0 capture

The tested capture project is shared with the Wave 2 source/corpus package at
[`experiments/quatops_capture_wave2`](../../../quatops_capture_wave2/LOCAL_REPLAY.md).
It pins `cesium@1.124.0`, the Cesium engine/widgets packages, and
`@zip.js/zip.js` in one lockfile with the Three.js/validator captures. Run
`npm ci` in that project's `js/` folder, then from the same folder use:

```powershell
node .\capture_cesium.mjs --output <new-output-directory>
```

The current frozen run is in
[`tests/golden/oracles/cesium/1.124.0`](../../../../tests/golden/oracles/cesium/1.124.0/manifest.json).
The SciPy replay and geographic-frame diagnostic are described in
[`docs/oracles/cesium.md`](../../../../docs/oracles/cesium.md).
