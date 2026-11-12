# glTF 2.0 transport fixtures

The assets exercise independent TRS and matrix transform forms, parent/child
composition, and `STEP`, `LINEAR`, and `CUBICSPLINE` quaternion animation on
nonuniform key times. The spec does not itself run a sampler; each named loader
has a distinct version directory and frozen observations.

## Reproducible asset

[`build_asset.py`](assets/build_asset.py) generates the two input files using
Python's standard library only. It imports no QuatOps, SciPy, or NumPy code.
The output reproduces the captured glTF and buffer byte hashes on every host;
the generated files are not checked in. The selected native observations and
their original input hashes remain in the loader fixtures.

Run the builder with explicit output paths from any working directory:

```shell
python experiments/transports/gltf/2.0.0/assets/build_asset.py \
  --binary-output experiments/transports/gltf/2.0.0/assets/rotation_transport.bin \
  --gltf-output experiments/transports/gltf/2.0.0/assets/rotation_transport.gltf
```

## Loaders

- [Three.js GLTFLoader 0.187.0](threejs/0.187.0/README.md)
- [Blender glTF importer 5.1.2](blender/5.1.2/README.md)

The Three.js and Blender records load the retained authored asset. A separate
QuatOps-generated static hierarchy is captured through Three.js for the
converter-boundary proof; see
[`experiments/converters/gltf/2.0.0`](../../../converters/gltf/2.0.0/README.md).
