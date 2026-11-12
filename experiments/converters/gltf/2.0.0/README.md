# glTF 2.0 static-hierarchy converter capture

This proof generates a two-node glTF hierarchy from explicit canonical input
frames, loads it with Three.js `GLTFLoader` 0.187.0, and retains the native
local/world actions needed to check serialization and hierarchy composition.
The loader's package and lockfile are kept in the separate versioned
[Three.js transport package](../../../transports/gltf/2.0.0/threejs/0.187.0/README.md).

From the repository root, run the cross-platform helper. It installs the
versioned loader dependencies, generates the asset, and captures selected native
node witnesses and provenance into the requested directory. Replay regenerates
the input and asset and checks both against their captured digests:

```shell
uv run --extra dev python -m experiments.run_converter_capture gltf --repository-root . --output <empty-capture-directory>
```

The helper requires Node.js 24 and npm on `PATH`; the npm lockfile pins the loader.
The capture accepts finite binary64 values and preserves signed zero in JSON.
Omit `--output` to create a persistent temporary capture directory. The
historical fixture is not overwritten.

The direct capture labels its point operation `Vector3.applyMatrix4`. The
historical record called this `Object3D.localToWorld`; the exporter actually
applied the world matrix to `Vector3` values.

The retained converter fixture is
[`tests/golden/transports/gltf/2.0.0/threejs/0.187.0/converter`](../../../../tests/golden/transports/gltf/2.0.0/threejs/0.187.0/converter/manifest.json)
and the replay is
[`tests/experiments/converters/gltf/test_gltf_converter_roundtrip.py`](../../../../tests/experiments/converters/gltf/test_gltf_converter_roundtrip.py).
