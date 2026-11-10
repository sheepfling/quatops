# glTF 2.0 static-hierarchy converter capture

This proof generates a two-node glTF hierarchy from explicit canonical input
frames, loads it with Three.js `GLTFLoader` 0.187.0, and retains the native
local/world actions needed to check serialization and hierarchy composition.
The loader's package and lockfile are kept in the separate versioned
[Three.js transport package](../../../transports/gltf/2.0.0/threejs/0.187.0/README.md).

From the repository root, run the cross-platform helper. It installs the
versioned loader dependencies, generates the asset, captures in a temporary
directory, and retains the selected native node witnesses. Replay regenerates
the input and asset and checks both against their captured digests:

```shell
uv run --extra dev python -m experiments.run_converter_capture gltf --repository-root .
```

The helper requires Node.js 24 and npm on `PATH`; the npm lockfile pins the loader.
The capture accepts finite binary64 values and preserves signed zero in JSON.
Older raw sidecar captures require the selector from their historical revision.

The retained converter fixture is
[`tests/golden/transports/gltf/2.0.0/threejs/0.187.0/converter`](../../../../tests/golden/transports/gltf/2.0.0/threejs/0.187.0/converter/manifest.json)
and the replay is
[`tests/experiments/converters/gltf/test_gltf_converter_roundtrip.py`](../../../../tests/experiments/converters/gltf/test_gltf_converter_roundtrip.py).
