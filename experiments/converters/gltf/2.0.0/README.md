# glTF 2.0 static-hierarchy converter capture

This proof generates a two-node glTF hierarchy from explicit canonical input
frames, loads it with Three.js `GLTFLoader` 0.187.0, and retains the native
local/world actions needed to check serialization and hierarchy composition.
The loader's package and lockfile are kept in the separate versioned
[Three.js transport package](../../../transports/gltf/2.0.0/threejs/0.187.0/README.md).

From the repository root, install the loader dependencies, generate the asset,
capture to a temporary directory, and select the two native node witnesses:

```powershell
npm ci --prefix experiments\transports\gltf\2.0.0\threejs\0.187.0
uv run python experiments\converters\gltf\2.0.0\create_asset.py
$captureDir = Join-Path $env:TEMP 'quatops-captures\gltf-static-0.187.0'
node experiments\converters\gltf\2.0.0\capture_threejs.mjs $captureDir
uv run python scripts\retain_gltf_converter_witnesses.py $captureDir
```

The retained converter fixture is
[`tests/golden/transports/gltf/2.0.0/threejs/0.187.0/converter`](../../../../tests/golden/transports/gltf/2.0.0/threejs/0.187.0/converter/manifest.json)
and the replay is
[`tests/experiments/converters/gltf/test_gltf_converter_roundtrip.py`](../../../../tests/experiments/converters/gltf/test_gltf_converter_roundtrip.py).
