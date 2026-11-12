# Authored glTF oriented-mesh witness

This bounded fixture adds an actual asymmetric tetrahedral mesh to the pinned
Three.js GLTFLoader evidence. The author declares vertex 1 to be the front
(+Z, distance 2), vertex 2 right (-X, distance 3), and vertex 3 up (+Y,
distance 5), relative to vertex 0. Unequal edges and noncommuting parent/child
rotations make front/back, axis-order and composition mistakes observable.

The geometry encodes an **application-declared asset alignment**. The glTF 2.0
oriented-asset convention is the source authority; this fixture does not prove
that arbitrary meshes comply with it. See the existing
[semantic source audit](../../../../../docs/semantic-source-audit.md).
The source QuatOps frame is explicitly +X right, +Y forward, +Z up. Its map to
this asset is `(-x, z, y)`, not the existing camera-like `(x, z, -y)` probe map.
Both are right-handed; their meanings differ.

## Reproduce

Install the existing pinned dependencies in `../threejs/0.187.0` using its
lockfile. From the repository root, run:

```shell
node experiments/transports/gltf/2.0.0/oriented_mesh/capture.mjs EMPTY_OUTPUT_DIRECTORY
python -m pytest tests/experiments/converters/gltf/test_oriented_mesh_evidence.py
```

The capture refuses to overwrite a nonempty output directory. It uses the
installed Three.js 0.187.0 GLTFLoader to read the embedded binary POSITION and
index buffers, observes actual mesh vertices, and records local, parent and
world coordinates. It preserves native observations; Python replay separately
checks those observations against SciPy matrices, authored geometry and the
QuatOps explicit frame map. No renderer or graphics context is required.

The manifest records the exact asset, capture script, dependency lock and
native-output SHA-256 values, along with Node/platform and library identity.
The initial native capture ran on Windows with Node 24.21.0 and Three.js
0.187.0. Replay is not another native run.

## Qualification boundary

This is additional, narrowly scoped loader-observation evidence for this
specified mesh, static hierarchy and authored alignment. The shared
semantic-claim registry links `gltf.asset-front` to this exact witness as
`asset_front_observation [oriented_mesh]`. Typed capture validation
checks geometry, author assignment, provenance and numeric actions before
crediting that limited claim. Converter acceptance is unchanged. It establishes neither
calibration of a physical installation nor inference of an unknown mesh's
front. Animation, arbitrary import/export pipelines, deformations, handedness
changes, negative/nonuniform scale and other loader versions are not covered.
