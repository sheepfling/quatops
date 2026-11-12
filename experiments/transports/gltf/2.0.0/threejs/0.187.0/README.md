# Three.js GLTFLoader 0.187.0

This is a named loader run over the reproducible glTF 2.0 assets in the sibling
`assets/` directory. It uses the exact Three.js package pin and Khronos
validator pin in this directory's `package-lock.json`.

## Reproduce

First [generate the input assets](../../README.md#reproducible-asset). Then,
from this directory:

```powershell
npm ci
node capture_loader.mjs --asset ..\..\assets\rotation_transport.gltf --output ..\..\..\..\..\..\tests\golden\transports\gltf\2.0.0\threejs\0.187.0\loader_rotation.json
```

The capture reads the `.gltf` and `.bin`, runs Khronos validation with the
external buffer bytes, embeds those same bytes as a data URI in memory for
Node's `GLTFLoader.parseAsync`, and records both original and adapted JSON
hashes. It adds a small `ProgressEvent` shim because Node does not supply that
browser constructor; no rendering or GPU route is used.

The validator reports zero errors, warnings, infos, or hints for this asset.
That is format/schema validation only. It does not prove animation sampling.
The named loader output separately records local/world row matrices, key
times, one discriminating midpoint per smooth interpolation, a STEP boundary
sample, and runtime/package/source hashes. Sampled key quaternions are retained
at the three authored key times; track values and duplicate matrix storage
views are omitted.
